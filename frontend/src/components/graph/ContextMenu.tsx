import React, { useEffect, useRef } from 'react';
import { GitBranch } from 'lucide-react';

interface ContextMenuItem {
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  variant?: 'default' | 'danger';
}

interface ContextMenuProps {
  x: number;
  y: number;
  nodeId: string;
  nodeName: string;
  nodeLabel: string;
  onShowRelations: (nodeId: string) => void;
  onClose: () => void;
}

export function ContextMenu({
  x,
  y,
  nodeId,
  nodeName,
  nodeLabel,
  onShowRelations,
  onClose
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  const items: ContextMenuItem[] = [
    {
      label: '展示相关联系',
      icon: <GitBranch className="w-4 h-4" />,
      onClick: () => {
        onShowRelations(nodeId);
        onClose();
      }
    }
  ];

  return (
    <div
      ref={menuRef}
      className="fixed bg-white rounded-lg shadow-xl border border-gray-200 py-2 min-w-[180px] z-50"
      style={{
        left: `${Math.min(x, window.innerWidth - 200)}px`,
        top: `${Math.min(y, window.innerHeight - 200)}px`
      }}
    >
      <div className="px-3 py-2 border-b border-gray-100">
        <div className="font-medium text-gray-900 text-sm truncate max-w-[200px]">
          {nodeName}
        </div>
        <div className="text-xs text-gray-500 flex items-center gap-1 mt-1">
          <span className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 rounded text-xs">
            {nodeLabel}
          </span>
        </div>
      </div>
      <div className="py-1">
        {items.map((item, index) => (
          <button
            key={index}
            onClick={item.onClick}
            className={`w-full px-3 py-2 text-left text-sm flex items-center gap-2 transition-colors ${
              item.variant === 'danger'
                ? 'text-red-600 hover:bg-red-50'
                : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            {item.icon}
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}
